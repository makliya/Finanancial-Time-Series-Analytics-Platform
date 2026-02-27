from django.contrib import admin

from .models import *

admin.site.register(User)


admin.site.site_header = '后台管理'  
admin.site.site_title = '管理员'  
admin.site.index_title = '3'